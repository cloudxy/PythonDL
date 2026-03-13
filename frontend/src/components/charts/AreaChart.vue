<template>
  <div class="w-full h-full">
    <Line v-if="loaded" :data="chartData" :options="chartOptions" />
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

const props = defineProps({
  title: {
    type: String,
    default: ''
  },
  labels: {
    type: Array,
    required: true
  },
  datasets: {
    type: Array,
    required: true
  },
  showLegend: {
    type: Boolean,
    default: true
  },
  showTooltip: {
    type: Boolean,
    default: true
  },
  gradient: {
    type: Boolean,
    default: true
  },
  smooth: {
    type: Boolean,
    default: true
  },
  height: {
    type: Number,
    default: 300
  }
})

const loaded = ref(false)

onMounted(() => {
  loaded.value = true
})

const chartData = computed(() => {
  const datasets = props.datasets.map((dataset, index) => ({
    ...dataset,
    fill: true,
    tension: props.smooth ? 0.4 : 0,
    borderColor: dataset.borderColor || getColor(index),
    pointRadius: dataset.pointRadius ?? 2,
    pointHoverRadius: dataset.pointHoverRadius ?? 5
  }))

  if (props.gradient) {
    datasets.forEach((dataset, index) => {
      dataset.backgroundColor = dataset.backgroundColor || getGradient(index)
    })
  }

  return {
    labels: props.labels,
    datasets
  }
})

const chartOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    title: {
      display: !!props.title,
      text: props.title,
      font: {
        size: 16,
        weight: 'bold'
      },
      padding: 20
    },
    legend: {
      display: props.showLegend,
      position: 'top',
      labels: {
        usePointStyle: true,
        pointStyle: 'circle'
      }
    },
    tooltip: {
      enabled: props.showTooltip,
      mode: 'index',
      intersect: false,
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      titleFont: {
        size: 14
      },
      bodyFont: {
        size: 13
      },
      padding: 12,
      cornerRadius: 8
    }
  },
  scales: {
    x: {
      grid: {
        display: false
      },
      ticks: {
        maxRotation: 45,
        minRotation: 45
      }
    },
    y: {
      grid: {
        color: 'rgba(0, 0, 0, 0.05)'
      },
      beginAtZero: false
    }
  },
  interaction: {
    mode: 'nearest',
    axis: 'x',
    intersect: false
  }
}))

function getColor(index) {
  const colors = [
    'rgba(59, 130, 246, 1)',
    'rgba(16, 185, 129, 1)',
    'rgba(245, 158, 11, 1)',
    'rgba(239, 68, 68, 1)',
    'rgba(139, 92, 246, 1)',
    'rgba(236, 72, 153, 1)'
  ]
  return colors[index % colors.length]
}

function getGradient(index) {
  const colorSets = [
    ['rgba(59, 130, 246, 0.5)', 'rgba(59, 130, 246, 0.0)'],
    ['rgba(16, 185, 129, 0.5)', 'rgba(16, 185, 129, 0.0)'],
    ['rgba(245, 158, 11, 0.5)', 'rgba(245, 158, 11, 0.0)'],
    ['rgba(239, 68, 68, 0.5)', 'rgba(239, 68, 68, 0.0)'],
    ['rgba(139, 92, 246, 0.5)', 'rgba(139, 92, 246, 0.0)'],
    ['rgba(236, 72, 153, 0.5)', 'rgba(236, 72, 153, 0.0)']
  ]
  return colorSets[index % colorSets.length]
}
</script>

<style scoped>
div {
  height: v-bind('height + "px"');
}
</style>
