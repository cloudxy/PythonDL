<template>
  <div class="w-full h-full">
    <Bar v-if="loaded" :data="chartData" :options="chartOptions" />
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { Bar } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js'

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
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
  horizontal: {
    type: Boolean,
    default: false
  },
  stacked: {
    type: Boolean,
    default: false
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

const chartData = computed(() => ({
  labels: props.labels,
  datasets: props.datasets.map((dataset, index) => ({
    ...dataset,
    backgroundColor: dataset.backgroundColor || getColor(index),
    borderColor: dataset.borderColor || getColor(index, 1),
    borderWidth: 1,
    borderRadius: 4,
    barThickness: dataset.barThickness || 'flex',
    maxBarThickness: dataset.maxBarThickness || 50
  }))
}))

const chartOptions = computed(() => ({
  indexAxis: props.horizontal ? 'y' : 'x',
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
        pointStyle: 'rect'
      }
    },
    tooltip: {
      enabled: props.showTooltip,
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
      stacked: props.stacked,
      ticks: {
        maxRotation: 45,
        minRotation: 45
      }
    },
    y: {
      grid: {
        color: 'rgba(0, 0, 0, 0.05)'
      },
      stacked: props.stacked,
      beginAtZero: true
    }
  }
}))

function getColor(index, alpha = 0.8) {
  const colors = [
    `rgba(59, 130, 246, ${alpha})`,
    `rgba(16, 185, 129, ${alpha})`,
    `rgba(245, 158, 11, ${alpha})`,
    `rgba(239, 68, 68, ${alpha})`,
    `rgba(139, 92, 246, ${alpha})`,
    `rgba(236, 72, 153, ${alpha})`,
    `rgba(14, 165, 233, ${alpha})`,
    `rgba(34, 197, 94, ${alpha})`
  ]
  return colors[index % colors.length]
}
</script>

<style scoped>
div {
  height: v-bind('height + "px"');
}
</style>
